"""Browser profile state — schema for cookies + per-origin storage."""

from __future__ import annotations

__all__ = [
    'ProfileState',
    'ProfileStateCookie',
    'ProfileStateIndexedDB',
    'ProfileStateIndexedDBIndex',
    'ProfileStateIndexedDBObjectStore',
    'ProfileStateIndexedDBRecord',
    'ProfileStateOriginStorage',
    'SameSitePolicy',
]

from collections.abc import Mapping, Sequence
from typing import Literal

import pydantic
import pydantic.alias_generators
from pydantic import JsonValue

from cc_lib.schemas.base import ClosedModel
from cc_lib.types import JsonObject

type SameSitePolicy = Literal['Strict', 'Lax', 'None']


class ProfileState(ClosedModel):
    """Browser profile state captured for session persistence.

    Cookies live at the top level because they use domain + path scoping —
    a single ``.example.com`` cookie applies to https://example.com,
    https://www.example.com, https://api.example.com, etc.

    Per-origin storage (localStorage, sessionStorage, IndexedDB) lives under
    ``origins`` keyed by origin string (``scheme://host:port``) because those
    APIs use strict same-origin scoping.

    The trailing optional fields (``extensions``, ``permissions``,
    ``preferences``) are reserved for future expansion.
    """

    schema_version: str = '1.0'
    captured_at: str | None = None  # ISO 8601 timestamp
    cookies: Sequence[ProfileStateCookie]
    origins: Mapping[str, ProfileStateOriginStorage]
    extensions: JsonObject | None = None
    permissions: JsonObject | None = None
    preferences: JsonObject | None = None


class ProfileStateCookie(ClosedModel):
    """One cookie in profile state format.

    Session cookies use ``expires=-1``; persistent cookies use an epoch
    timestamp. ``domain`` may be a leading-dot wildcard (``.example.com``)
    or an exact host.
    """

    name: str
    value: str
    domain: str
    path: str
    expires: float
    http_only: bool
    secure: bool
    same_site: SameSitePolicy


class ProfileStateOriginStorage(ClosedModel):
    """Storage data for a single origin.

    Each origin has isolated storage per the same-origin policy. All storage
    types for that origin are grouped here.

    ``session_storage`` is session-scoped by browser design; restored
    sessionStorage persists only for the lifetime of the browser context.
    For cross-session persistence, use cookies or ``local_storage``.
    """

    local_storage: Mapping[str, str]
    session_storage: Mapping[str, str] | None = None
    indexed_db: Sequence[ProfileStateIndexedDB] | None = None


class ProfileStateIndexedDB(ClosedModel):
    """IndexedDB database with version and object stores.

    Version is critical for schema migrations. Serializes to camelCase
    (``database_name → databaseName``, ``object_stores → objectStores``).
    """

    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    database_name: str
    version: int
    object_stores: Sequence[ProfileStateIndexedDBObjectStore]


class ProfileStateIndexedDBObjectStore(ClosedModel):
    """One object store within an IndexedDB database.

    Captures both schema (``key_path``, ``auto_increment``, ``indexes``)
    and data (``records``). Serializes to camelCase.
    """

    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    name: str
    key_path: str | Sequence[str] | None
    auto_increment: bool
    indexes: Sequence[ProfileStateIndexedDBIndex]
    records: Sequence[ProfileStateIndexedDBRecord]


class ProfileStateIndexedDBIndex(ClosedModel):
    """Index metadata on an IndexedDB object store.

    Serializes to camelCase (``key_path → keyPath``,
    ``multi_entry → multiEntry``).
    """

    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    name: str
    key_path: str | Sequence[str]
    unique: bool
    multi_entry: bool


class ProfileStateIndexedDBRecord(ClosedModel):
    """One record (key/value pair) in an IndexedDB object store.

    Keys can be strings, numbers, dates (as ISO strings), or arrays
    (compound keys). Values are JSON-serializable representations.

    Complex types are serialized with ``__type`` markers:

    - ``Date``: ``{"__type": "Date", "__value": "2024-01-01T00:00:00.000Z"}``
    - ``Map``: ``{"__type": "Map", "__value": [[key, value], ...]}``
    - ``Set``: ``{"__type": "Set", "__value": [item, ...]}``
    - ``ArrayBuffer``: ``{"__type": "ArrayBuffer", "__value": [byte, ...]}``
    """

    model_config = pydantic.ConfigDict(extra='forbid', strict=False)

    key: str | int | float | Sequence[str | int | float] | None
    value: JsonValue | None = None
