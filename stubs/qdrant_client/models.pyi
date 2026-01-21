"""Type stubs for qdrant_client.models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

class Distance(str, Enum):
    COSINE = 'Cosine'
    EUCLID = 'Euclid'
    DOT = 'Dot'
    MANHATTAN = 'Manhattan'

class VectorParams:
    size: int
    distance: Distance
    def __init__(self, size: int, distance: Distance, **kwargs: Any) -> None: ...

class PointStruct:
    id: str | int
    vector: Sequence[float]
    payload: Mapping[str, Any]
    def __init__(
        self,
        id: str | int,
        vector: Sequence[float],
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

class FieldCondition:
    key: str
    match: MatchAny | MatchText | MatchValue
    def __init__(
        self,
        key: str,
        match: MatchAny | MatchText | MatchValue,
        **kwargs: Any,
    ) -> None: ...

class Filter:
    must: Sequence[FieldCondition] | None
    should: Sequence[FieldCondition] | None
    must_not: Sequence[FieldCondition] | None
    def __init__(
        self,
        must: Sequence[FieldCondition] | None = None,
        should: Sequence[FieldCondition] | None = None,
        must_not: Sequence[FieldCondition] | None = None,
        **kwargs: Any,
    ) -> None: ...

class FilterSelector:
    filter: Filter
    def __init__(self, filter: Filter, **kwargs: Any) -> None: ...
