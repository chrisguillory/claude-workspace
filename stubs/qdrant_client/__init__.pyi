"""Type stubs for qdrant-client library."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qdrant_client.models import (
    Filter,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

class VectorsConfig:
    size: int

class VectorsParams:
    vectors: VectorsConfig | dict[str, VectorsConfig]

class CollectionConfig:
    params: VectorsParams

class CollectionInfo:
    name: str
    points_count: int | None
    config: CollectionConfig
    status: str

class CollectionsResponse:
    collections: Sequence[CollectionInfo]

class ScoredPoint:
    id: str | int
    score: float
    payload: dict[str, Any]

class Record:
    id: str | int
    payload: dict[str, Any]
    vector: Sequence[float] | None

class QueryResponse:
    points: Sequence[ScoredPoint]

class CountResult:
    count: int

class FacetHit:
    value: str | int | bool
    count: int

class FacetResponse:
    hits: Sequence[FacetHit]

class QdrantClient:
    def __init__(self, url: str | None = None, **kwargs: Any) -> None: ...
    def get_collections(self) -> CollectionsResponse: ...
    def get_collection(self, collection_name: str) -> CollectionInfo: ...
    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | Mapping[str, VectorParams] | None = None,
        sparse_vectors_config: Mapping[str, SparseVectorParams] | None = None,
        **kwargs: Any,
    ) -> bool: ...
    def upsert(
        self,
        collection_name: str,
        points: Sequence[PointStruct],
        **kwargs: Any,
    ) -> Any: ...
    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 10,
        score_threshold: float | None = None,
        query_filter: Filter | None = None,
        **kwargs: Any,
    ) -> list[ScoredPoint]: ...
    def query_points(
        self,
        collection_name: str,
        prefetch: Sequence[Any] | None = None,
        query: Any = None,
        limit: int = 10,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> QueryResponse: ...
    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[str | int],
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ) -> list[Record]: ...
    def delete(
        self,
        collection_name: str,
        points_selector: Sequence[str | int] | Any,
        **kwargs: Any,
    ) -> Any: ...
    def count(
        self,
        collection_name: str,
        count_filter: Filter | None = None,
        **kwargs: Any,
    ) -> CountResult: ...
    def scroll(
        self,
        collection_name: str,
        scroll_filter: Filter | None = None,
        limit: int = 10,
        offset: str | int | None = None,
        with_payload: bool | Sequence[str] = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ) -> tuple[list[Record], str | int | None]: ...
    def facet(
        self,
        collection_name: str,
        key: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> FacetResponse: ...
    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Any = None,
        **kwargs: Any,
    ) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
