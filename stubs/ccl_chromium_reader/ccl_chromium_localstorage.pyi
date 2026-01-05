"""Type stubs for ccl_chromium_reader.ccl_chromium_localstorage.

Partial stubs for LocalStoreDb context manager used to read Chrome localStorage.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

class LocalStorageRecord:
    """Record from localStorage LevelDB."""

    script_key: str
    value: Any

    def __getattr__(self, name: str) -> Any: ...

class LocalStoreDb:
    """Context manager for reading Chrome localStorage from LevelDB."""

    def __init__(self, path: Path) -> None: ...
    def __enter__(self) -> LocalStoreDb: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def iter_storage_keys(self) -> Iterator[str]: ...
    def iter_records_for_storage_key(self, storage_key: str) -> Iterator[LocalStorageRecord]: ...
    def __getattr__(self, name: str) -> Any: ...
