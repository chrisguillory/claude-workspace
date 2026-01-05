"""Type stubs for ccl_chromium_reader.ccl_chromium_sessionstorage.

Partial stubs for SessionStoreDb context manager used to read Chrome sessionStorage.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

class SessionStorageRecord:
    """Record from sessionStorage LevelDB."""

    key: str
    value: Any

    def __getattr__(self, name: str) -> Any: ...

class SessionStoreDb:
    """Context manager for reading Chrome sessionStorage from LevelDB."""

    def __init__(self, path: Path) -> None: ...
    def __enter__(self) -> SessionStoreDb: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def iter_hosts(self) -> Iterator[str]: ...
    def iter_records_for_host(self, host: str) -> Iterator[SessionStorageRecord]: ...
    def __getattr__(self, name: str) -> Any: ...
