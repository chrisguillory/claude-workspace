from __future__ import annotations

class Connection:
    id: str
    peername: tuple[str, int] | None
    sockname: tuple[str, int] | None
    timestamp_start: float | None
    timestamp_end: float | None
    timestamp_tls_setup: float | None
    tls_version: str | None
    sni: str | None
    @property
    def tls_established(self) -> bool: ...

class Client(Connection):
    peername: tuple[str, int]

class Server(Connection):
    address: tuple[str, int] | None
    timestamp_tcp_setup: float | None
