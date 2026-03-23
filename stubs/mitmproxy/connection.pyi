from __future__ import annotations

class Connection:
    id: str
    peername: tuple[str, int] | None
    sockname: tuple[str, int] | None
    timestamp_start: float | None
    timestamp_end: float | None

class Client(Connection):
    peername: tuple[str, int]

class Server(Connection):
    address: tuple[str, int] | None
    tls_established: bool
