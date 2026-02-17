"""Type stubs for qdrant-client library."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qdrant_client.models import (
    Filter,
    OptimizersConfigDiff,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

class VectorsConfig:
    size: int

class VectorsParams:
    vectors: VectorsConfig | dict[str, VectorsConfig]

class OptimizersConfig:
    indexing_threshold: int | None

class CollectionConfig:
    params: VectorsParams
    optimizer_config: OptimizersConfig

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

class AsyncQdrantClient:
    def __init__(self, url: str | None = None, **kwargs: Any) -> None: ...
    async def get_collections(self) -> CollectionsResponse: ...
    async def get_collection(self, collection_name: str) -> CollectionInfo: ...
    async def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | Mapping[str, VectorParams] | None = None,
        sparse_vectors_config: Mapping[str, SparseVectorParams] | None = None,
        **kwargs: Any,
    ) -> bool: ...
    async def update_collection(
        self,
        collection_name: str,
        optimizers_config: OptimizersConfigDiff | Any | None = None,
        **kwargs: Any,
    ) -> bool: ...
    async def delete_collection(
        self,
        collection_name: str,
        **kwargs: Any,
    ) -> bool: ...
    async def upsert(
        self,
        collection_name: str,
        points: Sequence[PointStruct],
        **kwargs: Any,
    ) -> Any: ...
    async def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 10,
        score_threshold: float | None = None,
        query_filter: Filter | None = None,
        **kwargs: Any,
    ) -> list[ScoredPoint]: ...
    async def query_points(
        self,
        collection_name: str,
        prefetch: Sequence[Any] | None = None,
        query: Any = None,
        limit: int = 10,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> QueryResponse: ...
    async def retrieve(
        self,
        collection_name: str,
        ids: Sequence[str | int],
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ) -> list[Record]: ...
    async def delete(
        self,
        collection_name: str,
        points_selector: Sequence[str | int] | Any,
        **kwargs: Any,
    ) -> Any: ...
    async def count(
        self,
        collection_name: str,
        count_filter: Filter | None = None,
        **kwargs: Any,
    ) -> CountResult: ...
    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Filter | None = None,
        limit: int = 10,
        offset: str | int | None = None,
        with_payload: bool | Sequence[str] = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ) -> tuple[list[Record], str | int | None]: ...
    async def facet(
        self,
        collection_name: str,
        key: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> FacetResponse: ...
    async def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Any = None,
        **kwargs: Any,
    ) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
