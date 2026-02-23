"""Type stubs for bashlex.errors."""

class ParsingError(Exception):
    message: str
    s: str
    position: int
    def __init__(self, message: str, s: str, position: int) -> None: ...
