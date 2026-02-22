"""Type stubs for bashlex — a Python port of bash's parser."""

from bashlex.ast import node

def parse(s: str) -> list[node]: ...
