"""Type stubs for rerankers library."""

from collections.abc import Sequence

class Result:
    doc_id: int
    score: float
    text: str
    rank: int

class RankedResults:
    results: Sequence[Result]
    query: str

class Reranker:
    def __init__(self, model_name: str, verbose: int = ...) -> None: ...
    def rank(self, query: str, docs: Sequence[str]) -> RankedResults: ...
