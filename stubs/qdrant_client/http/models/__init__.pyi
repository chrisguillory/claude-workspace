"""Type stubs for qdrant_client.http.models.

This is the canonical location for Qdrant model types. qdrant_client.models
re-exports these for convenience.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

class Distance(str, Enum):
    COSINE = 'Cosine'
    EUCLID = 'Euclid'
    DOT = 'Dot'
    MANHATTAN = 'Manhattan'

class Modifier(str, Enum):
    NONE = 'none'
    IDF = 'idf'

class Fusion(str, Enum):
    RRF = 'rrf'

class VectorParams:
    size: int
    distance: Distance
    def __init__(self, size: int, distance: Distance, **kwargs: Any) -> None: ...

class SparseVectorParams:
    modifier: Modifier | None
    def __init__(self, modifier: Modifier | None = None, **kwargs: Any) -> None: ...

class SparseVector:
    indices: Sequence[int]
    values: Sequence[float]
    def __init__(
        self,
        indices: Sequence[int],
        values: Sequence[float],
        **kwargs: Any,
    ) -> None: ...

# Vector type: dense, sparse, or multivector
type Vector = Sequence[float] | SparseVector | Sequence[Sequence[float]]

# VectorStruct: single vector or named vectors dict
type VectorStruct = Vector | Mapping[str, Vector]

class PointStruct:
    id: str | int
    vector: VectorStruct
    payload: Mapping[str, Any]
    def __init__(
        self,
        id: str | int,
        vector: VectorStruct,
        payload: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...

class MatchAny:
    any: Sequence[str]
    def __init__(self, any: Sequence[str], **kwargs: Any) -> None: ...

class MatchText:
    text: str
    def __init__(self, text: str, **kwargs: Any) -> None: ...

class MatchValue:
    value: str | int | bool
    def __init__(self, value: str | int | bool, **kwargs: Any) -> None: ...

# Condition union type - FieldCondition and other condition types
type Condition = (
    FieldCondition | IsEmptyCondition | IsNullCondition | HasIdCondition | HasVectorCondition | NestedCondition | Filter
)

class FieldCondition:
    key: str
    match: MatchAny | MatchText | MatchValue | None
    def __init__(
        self,
        key: str,
        match: MatchAny | MatchText | MatchValue | None = None,
        **kwargs: Any,
    ) -> None: ...

class IsEmptyCondition:
    is_empty: Any
    def __init__(self, is_empty: Any, **kwargs: Any) -> None: ...

class IsNullCondition:
    is_null: Any
    def __init__(self, is_null: Any, **kwargs: Any) -> None: ...

class HasIdCondition:
    has_id: Sequence[str | int]
    def __init__(self, has_id: Sequence[str | int], **kwargs: Any) -> None: ...

class HasVectorCondition:
    has_vector: str
    def __init__(self, has_vector: str, **kwargs: Any) -> None: ...

class NestedCondition:
    nested: Any
    def __init__(self, nested: Any, **kwargs: Any) -> None: ...

class Filter:
    must: Sequence[Condition] | Condition | None
    should: Sequence[Condition] | Condition | None
    must_not: Sequence[Condition] | Condition | None
    def __init__(
        self,
        must: Sequence[Condition] | Condition | None = None,
        should: Sequence[Condition] | Condition | None = None,
        must_not: Sequence[Condition] | Condition | None = None,
        **kwargs: Any,
    ) -> None: ...

class FilterSelector:
    filter: Filter
    def __init__(self, filter: Filter, **kwargs: Any) -> None: ...

class FusionQuery:
    fusion: Fusion
    def __init__(self, fusion: Fusion, **kwargs: Any) -> None: ...

class Prefetch:
    query: Sequence[float] | SparseVector | Any
    using: str | None
    limit: int | None
    filter: Filter | None
    def __init__(
        self,
        query: Sequence[float] | SparseVector | Any,
        using: str | None = None,
        limit: int | None = None,
        filter: Filter | None = None,
        **kwargs: Any,
    ) -> None: ...
