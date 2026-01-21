"""Type stubs for qdrant-client library."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client.models import Filter, PointStruct, VectorParams

class VectorsConfig:
    size: int

class VectorsParams:
    vectors: VectorsConfig

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

class QdrantClient:
    def __init__(self, url: str | None = None, **kwargs: Any) -> None: ...
    def get_collections(self) -> CollectionsResponse: ...
    def get_collection(self, collection_name: str) -> CollectionInfo: ...
    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams,
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
    def __getattr__(self, name: str) -> Any: ...
