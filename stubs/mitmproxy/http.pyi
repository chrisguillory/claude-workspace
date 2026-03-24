from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from mitmproxy import connection

class Headers:
    def __getitem__(self, key: str) -> str: ...
    def __contains__(self, key: object) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def get(self, key: str, default: str | None = None) -> str | None: ...
    def get_all(self, key: str) -> list[str]: ...
    def items(self) -> Iterator[tuple[str, str]]: ...

class MultiDictView[K, V]:
    def __getitem__(self, key: K) -> V: ...
    def get(self, key: K, default: V | None = None) -> V | None: ...
    def items(self) -> Iterator[tuple[K, V]]: ...
    def keys(self) -> Iterator[K]: ...
    def values(self) -> Iterator[V]: ...

class Request:
    method: str
    url: str
    host: str
    port: int
    path: str
    scheme: str
    headers: Headers
    content: bytes | None
    timestamp_start: float
    timestamp_end: float | None
    pretty_url: str
    http_version: str
    query: MultiDictView[str, str]
    cookies: MultiDictView[str, str]
    def get_text(self) -> str: ...

class Response:
    status_code: int
    reason: str
    headers: Headers
    content: bytes | None
    timestamp_start: float
    timestamp_end: float | None
    http_version: str
    cookies: MultiDictView[str, str]
    def get_text(self) -> str: ...

class HTTPFlow:
    id: str
    request: Request
    response: Response | None
    error: Error | None
    client_conn: connection.Client
    server_conn: connection.Server
    metadata: dict[str, Any]
    websocket: WebSocketData | None
    is_replay: str | None

class Error:
    msg: str
    timestamp: float

class WebSocketData:
    messages: list[WebSocketMessage]

class WebSocketMessage:
    type: int
    from_client: bool
    content: bytes
    timestamp: float
    is_text: bool
    text: str
