"""Type stubs for bashlex.ast."""

class node:
    kind: str
    pos: tuple[int, int]
    parts: list[node]
    list: list[node]
    word: str
    output: int | node
    type: str
