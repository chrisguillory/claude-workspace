"""IndexedDB Export with Full Schema using dfindexeddb.

This module replaces the ccl_chromium_reader-based IndexedDB export with
Google's dfindexeddb library, which provides complete schema extraction
(version, keyPath, autoIncrement, indexes).

The output format matches what indexeddb_restore.js expects for proper
database reconstruction in Selenium sessions.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dfindexeddb.indexeddb.chromium.definitions import (
    DatabaseMetaDataKeyType,
    IndexMetaDataKeyType,
    ObjectStoreMetaDataKeyType,
)
from dfindexeddb.indexeddb.chromium.record import (
    DatabaseMetaDataKey,
    DatabaseNameKey,
    FolderReader,
    IDBKeyPath,
    IndexMetaDataKey,
    ObjectStoreDataKey,
    ObjectStoreMetaDataKey,
)


def _extract_domain_from_origin(origin: str) -> str:
    """Extract domain from IndexedDB origin format.

    Chrome stores IndexedDB origins as directory names like:
    - https_example.com_0
    - http_localhost_3000_0

    Args:
        origin: Origin in Chrome's directory format

    Returns:
        Bare domain (e.g., "example.com", "localhost")
    """
    # Remove trailing _0, _1, etc. (partition key)
    if '_' in origin:
        parts = origin.rsplit('_', 1)
        if parts[-1].isdigit():
            origin = parts[0]

    # Remove scheme prefix (https_, http_, chrome-extension_, etc.)
    if '_' in origin:
        origin = origin.split('_', 1)[1]

    # Handle port (localhost_3000 -> localhost)
    if '_' in origin:
        parts = origin.rsplit('_', 1)
        if parts[-1].isdigit():
            origin = parts[0]

    return origin


def _domain_matches(host: str, pattern: str) -> bool:
    """RFC 6265 domain matching - suffix match with dot boundary.

    Args:
        host: Domain to check (e.g., "www.example.com")
        pattern: Pattern to match against (e.g., "example.com")

    Returns:
        True if host matches pattern per RFC 6265 rules
    """
    host = host.lower().strip('.')
    pattern = pattern.lower().strip('.')

    # Remove port from pattern if present
    if ':' in pattern:
        pattern = pattern.split(':')[0]

    # Exact match
    if host == pattern:
        return True

    # Suffix match with dot boundary
    return host.endswith('.' + pattern)


def _make_json_serializable(value: Any) -> Any:
    """Convert a value to JSON-serializable form.

    Handles special types from dfindexeddb that aren't directly serializable.
    Uses the same __type marker format as indexeddb_restore.js expects.

    Args:
        value: Any value that might need conversion

    Returns:
        JSON-serializable version of the value
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        # Encode as ArrayBuffer format matching indexeddb_restore.js deserializeValue()
        return {'__type': 'ArrayBuffer', '__value': list(value)}
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_serializable(v) for k, v in value.items()}
    # For other types, convert to string
    return str(value)


def _keypath_to_value(keypath: IDBKeyPath | None) -> str | list[str] | None:
    """Convert IDBKeyPath to JSON-serializable value.

    Args:
        keypath: IDBKeyPath object from dfindexeddb

    Returns:
        None for null keyPath, string for single path, list for array
    """
    if keypath is None:
        return None
    if keypath.value is None:
        return None
    return keypath.value


def _origin_dir_to_url(origin_dir: str) -> str:
    """Convert Chrome origin directory name to URL format.

    Args:
        origin_dir: Directory name like "https_example.com_0"

    Returns:
        URL like "https://example.com"
    """
    # Remove partition key suffix
    if '_' in origin_dir:
        parts = origin_dir.rsplit('_', 1)
        if parts[-1].isdigit():
            origin_dir = parts[0]

    # Parse scheme and domain
    if '_' in origin_dir:
        scheme, rest = origin_dir.split('_', 1)
        # Handle port if present (localhost_3000 -> localhost:3000)
        if '_' in rest:
            parts = rest.rsplit('_', 1)
            if parts[-1].isdigit():
                domain = parts[0]
                port = parts[1]
                return f'{scheme}://{domain}:{port}'
        return f'{scheme}://{rest}'

    # Fallback
    return f'https://{origin_dir}'


def export_indexeddb_with_schema(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Export IndexedDB with full schema using dfindexeddb.

    This function extracts complete IndexedDB data including:
    - Database version
    - Object store names, keyPaths, and autoIncrement settings
    - Index definitions (name, keyPath, unique, multiEntry)
    - All records

    Output uses snake_case keys matching ProfileStateIndexedDB Pydantic models.
    When serialized via Pydantic with by_alias=True, output converts to camelCase
    for JavaScript compatibility (e.g., database_name → databaseName).

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional domain patterns to include (e.g., ["amazon.com"])

    Returns:
        Dict mapping origin URLs to lists of database exports (snake_case):
        {
            "https://example.com": [
                {
                    "database_name": "mydb",
                    "version": 3,
                    "object_stores": [
                        {
                            "name": "users",
                            "key_path": "id",
                            "auto_increment": True,
                            "indexes": [
                                {
                                    "name": "email",
                                    "key_path": "email",
                                    "unique": True,
                                    "multi_entry": False
                                }
                            ],
                            "records": [
                                {"key": "1", "value": {...}}
                            ]
                        }
                    ]
                }
            ]
        }

    Raises:
        FileNotFoundError: If IndexedDB path doesn't exist
    """
    indexeddb_path = profile_path / 'IndexedDB'

    if not indexeddb_path.exists():
        return {}

    result: dict[str, list[dict[str, Any]]] = {}

    for db_dir in indexeddb_path.glob('*.indexeddb.leveldb'):
        origin_dir = db_dir.name.replace('.indexeddb.leveldb', '')

        # Filter by origin if specified
        if origins_filter:
            domain = _extract_domain_from_origin(origin_dir)
            if not any(_domain_matches(domain, pattern) for pattern in origins_filter):
                continue

        origin_url = _origin_dir_to_url(origin_dir)

        try:
            databases = _parse_indexeddb_folder(db_dir)
            if databases:
                result[origin_url] = databases
        except Exception as e:
            # Log but continue with other origins
            print(
                f'[indexeddb] Warning: Failed to parse {origin_dir}: {e}',
                file=sys.stderr,
            )

    return result


def _parse_indexeddb_folder(db_dir: Path) -> list[dict[str, Any]]:
    """Parse a single IndexedDB LevelDB folder.

    Args:
        db_dir: Path to .indexeddb.leveldb directory

    Returns:
        List of database exports with full schema
    """
    reader = FolderReader(db_dir)
    records = list(reader.GetRecords())

    # Collect metadata by database ID
    db_names: dict[int, str] = {}
    db_versions: dict[int, int] = {}

    # Object store metadata: {db_id: {store_id: {...}}}
    stores: dict[int, dict[int, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                'name': None,
                'keyPath': None,
                'autoIncrement': False,
                'indexes': {},
                'records': [],
            }
        )
    )

    # Index metadata: {db_id: {store_id: {index_id: {...}}}}
    indexes: dict[int, dict[int, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    # Process all records
    for rec in records:
        key = rec.key

        # Database name: key.database_name is the name, rec.value is the db_id
        if isinstance(key, DatabaseNameKey):
            db_id = rec.value
            if isinstance(db_id, int):
                db_names[db_id] = key.database_name

        # Database version
        elif isinstance(key, DatabaseMetaDataKey):
            if key.metadata_type == DatabaseMetaDataKeyType.IDB_INTEGER_VERSION:
                db_id = key.key_prefix.database_id
                # Keep the highest version seen
                if rec.value and (db_id not in db_versions or rec.value > db_versions[db_id]):
                    db_versions[db_id] = rec.value

        # Object store metadata
        elif isinstance(key, ObjectStoreMetaDataKey):
            db_id = key.key_prefix.database_id
            store_id = key.object_store_id
            store = stores[db_id][store_id]

            if key.metadata_type == ObjectStoreMetaDataKeyType.OBJECT_STORE_NAME:
                store['name'] = rec.value
            elif key.metadata_type == ObjectStoreMetaDataKeyType.KEY_PATH:
                store['keyPath'] = _keypath_to_value(rec.value)
            elif key.metadata_type == ObjectStoreMetaDataKeyType.AUTO_INCREMENT_FLAG:
                store['autoIncrement'] = bool(rec.value)

        # Index metadata
        elif isinstance(key, IndexMetaDataKey):
            db_id = key.key_prefix.database_id
            store_id = key.object_store_id
            index_id = key.index_id
            idx = indexes[db_id][store_id][index_id]

            if key.metadata_type == IndexMetaDataKeyType.INDEX_NAME:
                idx['name'] = rec.value
            elif key.metadata_type == IndexMetaDataKeyType.KEY_PATH:
                idx['keyPath'] = _keypath_to_value(rec.value)
            elif key.metadata_type == IndexMetaDataKeyType.UNIQUE_FLAG:
                idx['unique'] = bool(rec.value)
            elif key.metadata_type == IndexMetaDataKeyType.MULTI_ENTRY_FLAG:
                idx['multiEntry'] = bool(rec.value)

        # Object store data (records)
        elif isinstance(key, ObjectStoreDataKey):
            db_id = key.key_prefix.database_id
            store_id = key.key_prefix.object_store_id
            store = stores[db_id][store_id]

            # Extract key value from IDBKey
            record_key = key.encoded_user_key.value

            # Extract record value from ObjectStoreDataValue
            record_value = rec.value.value if hasattr(rec.value, 'value') else rec.value

            # Ensure values are JSON-serializable
            record_key = _make_json_serializable(record_key)
            record_value = _make_json_serializable(record_value)

            store['records'].append({'key': record_key, 'value': record_value})

    # Build output format matching ProfileStateIndexedDB Pydantic model
    result: list[dict[str, Any]] = []

    for db_id, db_name in db_names.items():
        if db_id not in stores:
            continue

        db_export: dict[str, Any] = {
            'database_name': db_name,
            'version': db_versions.get(db_id, 1),
            'object_stores': [],
        }

        for store_id, store_data in sorted(stores[db_id].items()):
            if store_data['name'] is None:
                continue

            # Add indexes to store (matching ProfileStateIndexedDBIndex)
            store_indexes = []
            if db_id in indexes and store_id in indexes[db_id]:
                for index_id, idx_data in sorted(indexes[db_id][store_id].items()):
                    if idx_data.get('name'):
                        store_indexes.append(
                            {
                                'name': idx_data['name'],
                                'key_path': idx_data.get('keyPath'),
                                'unique': idx_data.get('unique', False),
                                'multi_entry': idx_data.get('multiEntry', False),
                            }
                        )

            # Matching ProfileStateIndexedDBObjectStore
            store_export = {
                'name': store_data['name'],
                'key_path': store_data['keyPath'],
                'auto_increment': store_data['autoIncrement'],
                'indexes': store_indexes,
                'records': store_data['records'],
            }
            db_export['object_stores'].append(store_export)

        if db_export['object_stores']:
            result.append(db_export)

    return result
